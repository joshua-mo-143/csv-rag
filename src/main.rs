use std::io::{self, Write};

use csv::Reader;
use rig::{
    completion::Chat,
    embeddings::EmbeddingsBuilder,
    message::Message,
    vector_store::{
        in_memory_store::{InMemoryVectorIndex, InMemoryVectorStore},
        VectorStoreIndex,
    },
    Embed,
};

#[derive(serde::Deserialize, serde::Serialize, Debug, Eq, PartialEq)]
struct Record {
    first_name: String,
    last_name: String,
    email: String,
    role: String,
    salary: u32,
}

impl std::fmt::Display for Record {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            first_name,
            last_name,
            email,
            role,
            salary,
        } = self;
        write!(
            f,
            "First name: {first_name}\nLast name: {last_name}\nEmail: {email}\nRole: {role}\nSalary: {salary}"
        )
    }
}

impl Embed for Record {
    fn embed(
        &self,
        embedder: &mut rig::embeddings::TextEmbedder,
    ) -> Result<(), rig::embeddings::EmbedError> {
        Ok(embedder.embed(self.to_string()))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let openai_client = rig::providers::openai::Client::from_env();

    let agent = openai_client.agent("gpt-4o")
        .preamble("You are a helpful assistant. Your job is to answer a user's questions based on the context snippets given.").build();

    // Create embedding model
    let embedding_model = openai_client.embedding_model("text-embedding-ada-002");

    let index = load_csv(embedding_model).await?;

    println!(
        "Hi! This is your CSV ragger. Write a prompt and press Enter or write \"quit\" to exit. Alternatively, use \"reset\" to reset the conversation."
    );

    let mut chat_history: Vec<Message> = Vec::new();

    loop {
        print!("> ");
        let stdin = std::io::stdin();
        let mut prompt = String::new();
        io::stdout().flush()?;
        stdin.read_line(&mut prompt)?;

        match prompt.trim() {
            "quit" => break,
            "reset" => {
                chat_history.clear();
                println!("Your conversation has been reset.");
                continue;
            }
            _ => {}
        }

        let docs = index.top_n::<Record>(prompt.trim(), 4).await?;

        let prompt = format!(
            "Relevant employees:\n{}\n\n{}",
            docs.into_iter()
                .map(|(_, _, doc)| doc.to_string())
                .collect::<Vec<String>>()
                .join("\n"),
            prompt,
        );

        let answer = agent.chat(prompt.as_ref(), chat_history.clone()).await?;

        println!("{answer}");
        chat_history.push(Message::user(prompt));
        chat_history.push(Message::assistant(answer));
    }

    Ok(())
}

async fn load_csv(
    model: rig::providers::openai::EmbeddingModel,
) -> Result<
    InMemoryVectorIndex<rig::providers::openai::EmbeddingModel, Record>,
    Box<dyn std::error::Error>,
> {
    let reader = Reader::from_path("employees.csv")?;

    let documents: Vec<Record> = reader
        .into_deserialize::<Record>()
        .filter_map(|x| x.ok())
        .collect();

    let documents = match EmbeddingsBuilder::new(model.clone())
        .documents(documents)?
        .build()
        .await
    {
        Ok(ok) => ok,
        Err(e) => return Err(format!("Got error while embedding: {e}").into()),
    };

    let vector_store = InMemoryVectorStore::from_documents(documents);

    Ok(vector_store.index(model))
}
